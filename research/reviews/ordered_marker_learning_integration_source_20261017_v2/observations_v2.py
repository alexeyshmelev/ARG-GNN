"""Exact observed-only original loader and positive-plan scientific bridge.

One aligned route/plan per complete parent is shared by the two placement arms.
No generic cache framework, file format, execution wrapper or authority is added.
The original episode builder and F1 binder remain the authentication boundary.
"""
from __future__ import annotations
from dataclasses import is_dataclass
import sys,time
import numpy as np
import torch
from arggnn import train_ordered_clade_episode_v1 as h45
from arggnn import H040_latent_clade_execution_v1 as shared
from arggnn.mutation_routed_full_panel_fast_reader_v1 import (
    FullPanelParentDatasetV1,make_parent_dataloader_v1,shutdown_parent_dataloader_v1)
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.cpu_cache_v1 import (
    FrozenEpisodeCacheV1,cache_preflight_v1)
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.observations_v1 import (
    _prepare_payload_once_v1,tensor_bytes,device_parent_v1)
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.protocol_v1 import ARMS as ORIGINAL_H058_ARMS
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.observation_binding_v3 import bank_stamps,episode_stamps
from .protocol_v2 import TRAIN,VAL,ARMS,OBSERVATION,require
from .positive_marker_backend_v2 import prepare_positive_parent_v2
from .ordered_learning_gnn_v2 import attach_positive_plan_v2


def retained_size_v2(roots):
    """Count owned Python objects and union CPU storage intervals once.

    This is preparation accounting, never a fit/peak-memory certificate. Source
    ownership still forbids foreign writers or untracked aliases. Native library
    workspaces and temporary allocator peaks remain under ROOT's real RSS bound.
    """
    seen=set();intervals=[];objects=0
    def visit(value):
        nonlocal objects
        if id(value) in seen:return
        seen.add(id(value))
        size=sys.getsizeof(value)
        if isinstance(value,np.ndarray):
            # ndarray.__sizeof__ includes owned backing memory; count it in the
            # storage interval union, not twice as Python object overhead.
            objects+=size-(value.nbytes if value.flags.owndata else 0)
            base=value
            while isinstance(base.base,np.ndarray):base=base.base
            if base.nbytes:
                start=int(base.__array_interface__["data"][0])
                intervals.append((start,start+base.nbytes))
            if value.base is not None:visit(value.base)
            return
        objects+=size
        if isinstance(value,torch.Tensor):
            require(value.device.type=='cpu','Preparation storage must remain CPU-only')
            storage=value.untyped_storage()
            if storage.nbytes():intervals.append((storage.data_ptr(),storage.data_ptr()+storage.nbytes()))
        elif isinstance(value,dict):
            for key,item in value.items():visit(key);visit(item)
        elif isinstance(value,(tuple,list,set,frozenset)):
            for item in value:visit(item)
        elif is_dataclass(value) or hasattr(value,'__dict__'):
            visit(vars(value))
    visit(roots)
    total=0;left=right=None
    for start,stop in sorted(intervals):
        if right is None:left,right=start,stop
        elif start<=right:right=max(right,stop)
        else:total+=right-left;left,right=start,stop
    if right is not None:total+=right-left
    return dict(numeric_storage_bytes=total,Python_object_bytes=objects,total_bytes=total+objects)


def load_observations_v2(ctx,reader,ids):
    require(tuple(ids) in (TRAIN,VAL),'Exact complete original TRAIN or VAL corpus')
    require(ctx['observation_cache_ref']==OBSERVATION,'Frozen original observations required')
    forecast=cache_preflight_v1(ctx,ids) # Original two-route upper remains conservative for one route.
    dataset=FullPanelParentDatasetV1(OBSERVATION,registry=ctx['registry'],
        entry_refs=ctx['entry_refs'],source_refs=ctx['observation_source_refs'],
        parent_ids=ids,worker_tmp_directory=ctx['worker_tmp_directory'],allow_outer=False)
    loader=make_parent_dataloader_v1(dataset,pin_memory=False)
    episodes=FrozenEpisodeCacheV1(ctx['cfg'],ctx['guard'],ctx['before_episode_build'])
    banks,parents,keys,rosters,proofs,site_refs={},{},{},{},{},{}
    total=0
    try:
        for expected,item in zip(ids,loader,strict=True):
            ctx['guard']()
            require(item.parent_id==expected and item.worker_cuda_visible_devices==''
                and item.worker_cpu_threads==1,'Original ordered two-worker CPU loader')
            bank=item.to_model_inputs()
            require(item.scoring['split']==('train' if expected in TRAIN else 'val')
                and item.observations_receipt['targets_read'] is False
                and bank.sample_index.shape[1]==952 and bank.query_column.numel()==686,
                'Every original node/hold/site/sample/query remains')
            total+=tensor_bytes(bank)
            require(total<=ctx['cfg']['CPU_bank_cap_bytes'],'Complete original CPU bank cap')
            banks[expected],parents[expected]=bank,item.scoring
            keys[expected]=shared.stable_keys(bank,ctx['entry_refs'][expected],reader)
            entry=reader.decode_json(ctx['entry_refs'][expected])
            site_refs[expected]=entry['refs']['site_coordinates']
            reader.decode_json(site_refs[expected])
            rosters[expected]=reader.decode_json(entry['refs']['source_roster'])
            episode=episodes.get(expected,bank,keys[expected])
            proofs[expected]=dict(observed_entry_ref=ctx['entry_refs'][expected],
                H045_episode=episodes.proofs[expected],native_holds=int(bank.sample_index.shape[0]),
                native_nodes=bank.parent.numel(),original_SNPs=len(bank.genotype),
                original_haplotypes=952,original_queries=686,targets_read=False)
            del episode,item,bank
    finally:
        drain=shutdown_parent_dataloader_v1(loader)
        require(drain['drained'] is True and all(code==0 for code in drain['worker_exitcodes']),
            'Original DataLoader actual shutdown/waits')
    require(set(banks)==set(ids),'All original phase observations precede truth')
    episodes.freeze_v1(banks,keys,ids)
    observed=dict(banks=banks,parents=parents,keys=keys,rosters=rosters,episodes=episodes,
        proofs=proofs,site_refs=site_refs,payloads={},packing={},identities={},forecast=forecast)
    required_count_bytes=256*sum(len(bank.genotype) for bank in banks.values())
    require(required_count_bytes<=ctx['cfg']['CPU_positive_plan_cap_bytes'],
        'All original per-site called-count tensors must fit; no reduced-support fallback')
    packed_bytes=packed_objects=0
    for pid in ids:
        ctx['guard']();bank=banks[pid];episode=episodes.get(pid,bank,keys[pid]);started=time.monotonic()
        baseline,proof=_prepare_payload_once_v1(ctx,observed,pid,ORIGINAL_H058_ARMS[0],bank,episode)
        plan=prepare_positive_parent_v2(baseline.route_cpu,bank,runtime_check=ctx['guard'])
        payload=attach_positive_plan_v2(baseline,plan)
        observed['payloads'][pid]=payload
        observed['identities'][pid]=(id(bank),bank_stamps(bank),id(episode),episode_stamps(episode),id(payload))
        packed_bytes+=payload.route_cpu.numeric_tensor_bytes
        packed_objects+=payload.route_cpu.tensor_object_count
        require(packed_bytes<=ctx['cfg']['CPU_packed_cache_cap_bytes']
            and packed_objects<=ctx['cfg']['CPU_packed_cache_max_tensor_objects'],
            'Original packed-route aggregate caps remain')
        positive=retained_size_v2(tuple(p.positive_plan for p in observed['payloads'].values()))
        require(positive['total_bytes']<=ctx['cfg']['CPU_positive_plan_cap_bytes'],
            'Complete positive plan including Python objects exceeds reviewed cap')
        retained=retained_size_v2((observed,episodes.values))
        require(retained['total_bytes']+forecast['target_bytes']<=ctx['cfg']['CPU_prepared_cache_cap_bytes'],
            'All complete prepared parents plus original target allowance must fit; no eviction/spill/rebuild')
        observed['packing'][pid]=dict(proof,seconds=time.monotonic()-started,
            placement_arms_share_exact_same_aligned_payload=True,positive_plan=positive,
            original_blocks=sum((len(h.covers)+15)//16 for h in plan.holds),
            forced_dense_blocks=sum(any(h.prefer_dense[s:s+16]) for h in plan.holds
                for s in range(0,len(h.covers),16)),retained_preparation=retained)
        del baseline,plan,payload,bank,episode
    require(set(observed['payloads'])==set(ids) and episodes.builds==len(ids),
        'All original episodes/routes/positive plans built once before phase truth')
    final_retained=retained_size_v2((observed,episodes.values))
    require(final_retained['total_bytes']+forecast['target_bytes']<=ctx['cfg']['CPU_prepared_cache_cap_bytes'],
        'Complete bridge metadata, numerical storage and target allowance exceed existing aggregate cap')
    observed['sealed']=True
    observed['seal']=ctx['output'].save('all_%d_ordered_marker_observation_recipes.json'%len(ids),dict(
        schema='ordered_marker_full_observation_barrier_v2',parent_ids=list(ids),recipes=proofs,
        arms=list(ARMS),observation_cache_ref=OBSERVATION,original_CPU_DataLoader_workers=2,
        worker_drain=drain,targets_read=False,all_original_trees_sites_haplotypes_queries_retained=True,
        one_aligned_payload_per_parent_shared_between_placement_arms=True,
        positive_preparation={p:observed['packing'][p] for p in ids},
        original_two_route_preflight_upper=forecast,retained_preparation=final_retained,measured_training_fit=False))
    ctx['_ordered_marker_observed']=observed
    return observed


def payload_v2(ctx,observed,pid,arm,bank,episode):
    require(observed.get('sealed') is True and arm in ARMS and pid in observed['payloads'],
        'Only complete pretruth prepared original payloads')
    observed['episodes'].validate_active_v1(pid,bank,observed['keys'][pid])
    payload=observed['payloads'][pid]
    require((id(bank),bank_stamps(bank),id(episode),episode_stamps(episode),id(payload))
        ==observed['identities'][pid],'Active observed bank/episode/payload changed')
    payload.binding.validate(bank,episode,payload.route_cpu)
    ctx['_ordered_marker_active']=(pid,id(bank),id(payload))
    return payload,dict(observed['packing'][pid],CPU_payload_same_object=True,
        cached_preparation_seconds=observed['packing'][pid]['seconds'],seconds=0.)


def validate_scheduled_v2(ctx,bank,payload):
    observed=ctx['_ordered_marker_observed'];pid,bank_id,payload_id=ctx['_ordered_marker_active']
    require(observed['sealed'] and id(bank)==bank_id and id(payload)==payload_id
        and observed['payloads'][pid] is payload,'Scheduled original payload identity changed')
    observed['episodes'].validate_active_v1(pid,bank,observed['keys'][pid])
    payload.binding.validate(bank,payload.episodes,payload.route_cpu)


def scheduled_CPU_receipt_v2(ctx):
    observed=ctx['_ordered_marker_observed']
    return dict(parents=len(observed['payloads']),episode_builds=observed['episodes'].builds,
        all_preparation_before_truth=observed['sealed'],two_arms_share_original_aligned_route=True,
        streamed_marker_evidence=True,per_marker_block_checkpoint=16,
        full_parent_fit_measured_by_actual_scheduled_step_only=True)

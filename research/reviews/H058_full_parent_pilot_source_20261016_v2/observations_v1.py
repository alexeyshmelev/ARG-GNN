"""Existing full-parent PyG loader, original H045 episodes, exact packed/F1 setup."""
from __future__ import annotations
from dataclasses import fields
import hashlib,json,time
import torch
from arggnn import train_ordered_clade_episode_v1 as h45
from arggnn import H040_latent_clade_execution_v1 as shared
from arggnn.mutation_routed_full_panel_fast_reader_v1 import (
    FullPanelParentDatasetV1, make_parent_dataloader_v1, shutdown_parent_dataloader_v1)
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import prepare_packed_parent_v2
from research.reviews.biological_operator_next_fresh_20261016_v1.source_handoff.same_marker_allele_support_v1 import build_alignment_null_v1
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.observation_binding_v3 import (
    CORE_INPUTS, bind_current_observations_v3, validate_observed_roster_v2)
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.integration_hooks_v3 import PackedSameMarkerPayloadV3
from .protocol_v1 import ARMS, CONTINUATION, OBSERVATION, TRAIN, VAL, require


def tensor_bytes(value):
    return sum(t.numel()*t.element_size() for f in fields(value)
        for t in (getattr(value,f.name) if isinstance(getattr(value,f.name),tuple) else (getattr(value,f.name),)))


def identity(value):
    raw=value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def null_for_parent(bank, roster):
    ids={}
    diploid=torch.tensor([ids.setdefault(r["sample_id"],len(ids)) for r in roster],dtype=torch.long)
    phase=torch.tensor([r["haplotype"] for r in roster],dtype=torch.long)
    permutation,receipt=build_alignment_null_v1(bank.reference_class,bank.sample_age,diploid,phase,seed=101)
    validate_observed_roster_v2(roster,bank,permutation)
    groups=4*(bank.sample_age>=350).long()+bank.reference_class
    summaries=[]
    for group in range(8):
        columns=torch.nonzero((bank.reference_class>=0)&(groups==group),as_tuple=False).flatten()
        moved=columns[permutation[columns]!=columns]
        called=moved_called=0
        # Bounded observed bookkeeping; no model/feature evaluation or target.
        for start in range(0,len(bank.genotype),256):
            block=bank.genotype[start:start+256]
            left=block.index_select(1,columns)
            right=block.index_select(1,permutation[columns])
            for allele in (-1,0,1,2,3):
                require(torch.equal(left.eq(allele).sum(1),right.eq(allele).sum(1)),
                    "Frozen null preserves each marker's source/age allele and call count")
            called+=int(left.ge(0).sum())
            moved_called+=int(block.index_select(1,moved).ge(0).sum())
        summaries.append(dict(group=group,references=len(columns),moved_haplotypes=len(moved),
            called_entries=called,moved_called_entries=moved_called,
            immovable_group=len(columns)<=2))
    receipt.update(permutation_sha256=identity(permutation),called_moved_mass_not_computed=False,
        groups=summaries,all_original_SNPs_bookkeeping_only=True)
    return permutation,receipt


def load_observations_v1(ctx,reader,ids):
    """Two original CPU spawn workers; one complete parent, never batching or crop.

    CPU banks are retained under the original aggregate RAM cap. Dataset workers
    use original observed-only readers and Linux abstract IPC, never phase truth.
    Original workers are actually shut down before the observation seal.
    """
    require(tuple(ids) in (TRAIN,VAL), "Exact complete TRAIN or VAL corpus")
    require(ctx["observation_cache_ref"]==OBSERVATION, "Frozen original observation bank")
    dataset=FullPanelParentDatasetV1(OBSERVATION,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"],
        parent_ids=ids,worker_tmp_directory=ctx["worker_tmp_directory"],allow_outer=False)
    loader=make_parent_dataloader_v1(dataset,pin_memory=False)
    episodes=h45.EpisodeCache(ctx["cfg"],ctx["guard"],ctx["before_episode_build"])
    banks,parents,keys,rosters,permutations,proofs,site_refs={},{},{},{},{},{},{}
    total=0
    drain=None
    try:
        for expected,item in zip(ids,loader,strict=True):
            ctx["guard"]()
            require(item.parent_id==expected and item.worker_cuda_visible_devices==""
                and item.worker_cpu_threads==1, "Original ordered CPU-only two-worker Dataset")
            bank=item.to_model_inputs()
            require(item.scoring["split"]==("train" if expected in TRAIN else "val")
                and item.observations_receipt["targets_read"] is False
                and tuple(bank.sample_index.shape)[1]==952 and bank.query_column.numel()==686,
                "Every original native hold, sample and query retained")
            total+=tensor_bytes(bank)
            require(total<=ctx["cfg"]["CPU_bank_cap_bytes"], "Full retained CPU bank aggregate cap")
            banks[expected],parents[expected]=bank,item.scoring
            keys[expected]=shared.stable_keys(bank,ctx["entry_refs"][expected],reader)
            entry=reader.decode_json(ctx["entry_refs"][expected])
            site_refs[expected]=entry["refs"]["site_coordinates"]
            reader.decode_json(site_refs[expected])  # Authenticated observed SNP coordinates only.
            roster=reader.decode_json(entry["refs"]["source_roster"])
            rosters[expected]=roster
            permutations[expected],null_receipt=null_for_parent(bank,roster)
            episode=episodes.get(expected,bank,keys[expected])
            proofs[expected]=dict(observed_entry_ref=ctx["entry_refs"][expected],
                H045_episode=episodes.proofs[expected],null=null_receipt,
                native_holds=int(bank.sample_index.shape[0]),native_nodes=bank.parent.numel(),
                original_SNPs=len(bank.genotype),CPU_bank_bytes=tensor_bytes(bank),
                original_haplotypes=952,original_queries=686,targets_read=False)
            del episode,item,bank
    finally:
        drain=shutdown_parent_dataloader_v1(loader)
        require(drain["drained"] is True
            and all(code==0 for code in drain["worker_exitcodes"]),
            "Actual original CPU DataLoader shutdown/waits required")
    require(set(banks)==set(ids), "All original phase parents before truth")
    seal=ctx["output"].save("all_%d_H058_observation_recipes.json"%len(ids),dict(
        schema="H058_full_observation_barrier_v1",hypothesis_id="H058",parent_ids=list(ids),
        recipes=proofs,observation_cache_ref=OBSERVATION,CPU_bank_tensor_bytes=total,
        original_CPU_DataLoader_workers=2,worker_drain=drain,targets_read=False,
        all_original_trees_sites_haplotypes_queries_retained=True))
    return dict(banks=banks,parents=parents,keys=keys,rosters=rosters,
        permutations=permutations,episodes=episodes,seal=seal,proofs=proofs,site_refs=site_refs)


def device_parent_v1(ctx,observed,pid,mass=None):
    ctx["guard"]()
    bank=observed["banks"][pid]
    episode=observed["episodes"].get(pid,bank,observed["keys"][pid])
    resident=tensor_bytes(bank)+episode.tensor_bytes()+(0 if mass is None else mass.numel()*mass.element_size())
    require(episode.tensor_bytes()<=ctx["cfg"]["GPU_episode_cap_bytes"]
        and resident<=ctx["cfg"]["GPU_residency_cap_bytes"], "One shared complete original GPU bank/episode")
    return bank.to(ctx["device"]),episode.to(ctx["device"]),None if mass is None else mass.to(ctx["device"])


def payload_v1(ctx,observed,pid,arm,device_bank,device_episode):
    """At most ONE parent/ONE arm packed layout alive. No cross-parent route cache."""
    require(arm in (*ARMS,CONTINUATION), "Registered arm")
    if arm==CONTINUATION:
        return device_episode,None
    started=time.monotonic()
    cpu_bank=observed["banks"][pid]
    cpu_episode=observed["episodes"].get(pid,cpu_bank,observed["keys"][pid])
    inputs={key:getattr(cpu_bank,key) for key in CORE_INPUTS}
    if arm==ARMS[1]:
        inputs["route_permutation"]=observed["permutations"][pid]
    route=prepare_packed_parent_v2(**inputs,
        max_layout_tensor_bytes=ctx["cfg"]["packed_layout_cap_bytes"],
        max_prepare_named_array_bytes=ctx["cfg"]["packed_prepare_named_cap_bytes"])
    require(max(hold.nodes for hold in route.holds)<=ctx["cfg"]["packed_max_hold_nodes"],
        "Full observed hold exceeds reviewed activation bound; fail without dropping data")
    binding=bind_current_observations_v3(route=route,route_inputs_cpu=inputs,
        cpu_bank=cpu_bank,device_bank=device_bank,cpu_episode=cpu_episode,
        device_episode=device_episode,observed_roster=observed["rosters"][pid],
        observed_entry_ref=ctx["entry_refs"][pid],observed_cache_ref=OBSERVATION,
        site_sidecar_ref=observed["site_refs"][pid],
        root_full_parent_layout_resource_proof_ref=ctx["cfg"]["layout_resource_proof_ref"],
        chunk_elements=ctx["cfg"]["binding_chunk_elements"])
    proof=dict(parent_id=pid,arm=arm,packed_numeric_bytes=route.numeric_tensor_bytes,
        packed_tensor_objects=route.tensor_object_count,
        named_preparation_allowance_bytes=route.preparation_named_array_allowance_bytes,
        binding_comparison_scratch_allowance_bytes=24*2**20,
        seconds=time.monotonic()-started,holds=len(route.holds),
        actual_max_hold_nodes=max(h.nodes for h in route.holds),
        GPU_packed_whole_parent_copy=False,CPU_route_cache_parents=1)
    return PackedSameMarkerPayloadV3(device_episode,route,binding),proof

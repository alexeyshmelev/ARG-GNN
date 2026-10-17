"""H058 complete CPU preparation reused with one transient CUDA parent.
No admission, resource discovery, route rebuilding, target selection or model change.
"""
from __future__ import annotations
import time
import torch
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1 import observations_v1 as cpu
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.protocol_v1 import ARMS,TRAIN,require

tensor_bytes=cpu.tensor_bytes
load_observations_v1=cpu.load_observations_v1

def device_parent_v1(ctx,observed,pid,mass=None):
    ctx["guard"]()
    require(ctx["device"].type=="cuda" and ctx["physical_gpu"]==0
        and pid in TRAIN and "_H058_GPU_active" not in ctx,
        "One complete CUDA parent at a time; previous pair explicitly released")
    bank=observed["banks"][pid]
    episode=observed["episodes"].get(pid,bank,observed["keys"][pid])
    resident=tensor_bytes(bank)+episode.tensor_bytes()+(0 if mass is None else mass.numel()*mass.element_size())
    require(episode.tensor_bytes()<=ctx["cfg"]["GPU_episode_cap_bytes"]
        and resident<=ctx["cfg"]["GPU_residency_cap_bytes"],
        "Original complete bank/episode/mass residency cap")
    require(mass is None or(mass.device.type=="cpu" and mass.dtype==torch.float32),
        "Original complete CPU FP32 target mass only")
    started=time.monotonic()
    device_bank=bank.to(ctx["device"])
    device_episode=episode.to(ctx["device"])
    device_mass=None if mass is None else mass.to(ctx["device"])
    torch.cuda.synchronize(ctx["device"])
    ctx["_H058_GPU_active"]=dict(pid=pid,observed=observed,cpu_bank=bank,cpu_episode=episode,
        bank=device_bank,episode=device_episode,mass=device_mass,bound=None,
        scheduled_arms=[],transfer_seconds=time.monotonic()-started,
        original_resident_bytes=resident,bindings={})
    ctx["guard"]()
    return device_bank,device_episode,device_mass

def payload_v1(ctx,observed,pid,arm,device_bank,device_episode):
    a=ctx["_H058_GPU_active"]
    require(a["pid"]==pid and a["observed"] is observed and a["bank"] is device_bank
        and a["episode"] is device_episode and arm in ARMS and arm not in a["bindings"],
        "Exactly one fresh device binding per original arm and resident parent")
    cpu_payload,proof=observed["cpu_cache"].get_v1(pid,arm,a["cpu_bank"],a["cpu_episode"])
    inputs={key:getattr(a["cpu_bank"],key) for key in cpu.CORE_INPUTS}
    if arm==ARMS[1]:inputs["route_permutation"]=observed["permutations"][pid]
    started=time.monotonic()
    binding=cpu.bind_current_observations_v3(route=cpu_payload.route_cpu,route_inputs_cpu=inputs,
        cpu_bank=a["cpu_bank"],device_bank=device_bank,cpu_episode=a["cpu_episode"],
        device_episode=device_episode,observed_roster=observed["rosters"][pid],
        observed_entry_ref=ctx["entry_refs"][pid],observed_cache_ref=cpu.OBSERVATION,
        site_sidecar_ref=observed["site_refs"][pid],
        root_full_parent_layout_resource_proof_ref=ctx["cfg"]["layout_resource_proof_ref"],
        chunk_elements=ctx["cfg"]["binding_chunk_elements"])
    payload=cpu.PackedSameMarkerPayloadV3(device_episode,cpu_payload.route_cpu,binding)
    a["bound"]=(arm,cpu_payload,payload)
    a["bindings"][arm]=dict(seconds=time.monotonic()-started,
        fresh_device_F1_binding=True,CPU_binding_reused_as_GPU=False,
        CPU_route_same_object=True,GPU_whole_route_copy=False)
    proof=dict(proof,device_transfer_seconds_once_per_pair=a["transfer_seconds"],
        device_binding=a["bindings"][arm],CPU_preparation_before_TRAIN_truth=True,
        scheduled_device_transfer_and_binding_use_observations_only=True)
    return payload,proof

def validate_scheduled_cache_v1(ctx,bank,payload):
    a=ctx["_H058_GPU_active"];arm,cpu_payload,bound=a["bound"]
    require(a["bank"] is bank and bound is payload and payload.episodes is a["episode"]
        and payload.route_cpu is cpu_payload.route_cpu,"Exact current CUDA parent/arm payload")
    a["observed"]["cpu_cache"].validate_scheduled_v1(a["cpu_bank"],cpu_payload)
    payload.binding.validate(bank,a["episode"],payload.route_cpu)

def scheduled_GPU_receipt_v1(ctx):
    a=ctx["_H058_GPU_active"];arm,_,_=a["bound"]
    return dict(CPU_preparation=cpu.scheduled_CPU_receipt_v1(ctx),
        parent_id=a["pid"],device_transfer_seconds_once_per_pair=a["transfer_seconds"],
        device_binding=dict(a["bindings"][arm]),original_resident_bytes=a["original_resident_bytes"],
        one_parent_device_resident=True,two_arms_sequential=True,
        extra_model_probe_or_optimizer_update=False,retained_in_full960=True)

def release_device_parent_v1(ctx,bank,episode,mass):
    a=ctx["_H058_GPU_active"]
    require(a["bank"] is bank and a["episode"] is episode and a["mass"] is mass
        and set(a["bindings"])==set(ARMS),"Both full arm updates before resident-parent release")
    a["bound"]=None
    del ctx["_H058_GPU_active"]

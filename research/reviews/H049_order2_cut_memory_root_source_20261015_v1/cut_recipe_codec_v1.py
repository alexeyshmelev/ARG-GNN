"""Lossless compact H049 cut recipes; frozen model science is unchanged."""
from __future__ import annotations
import io
import math
from dataclasses import replace

FORMAT = "H049_distinct_cut_uint16_uint32_packbits_v1"
MEMBERS = ("representatives_u32", "episode_groups_u16", "episode_of_hold_u32",
           "group_offsets_u32", "above_horizon_bits", "root_saturated_bits")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact_upper_v1(n, t, h=952, *, groups=None, episodes=None):
    require(all(type(x) is int and x > 0 for x in (n,t,h)) and h==952,
            "Positive complete native counts")
    k = min(n,t*h) if groups is None else groups
    g = t if episodes is None else episodes
    require(type(k) is int and 0<k<=min(n,t*h) and type(g) is int and 0<g<=t,
            "Verified compact counts")
    raw = 4*k+2*g*h+4*t+4*(t+1)+(h+7)//8+(k+7)//8
    return dict(compact_raw_array_bytes=raw,
                compressed_NPZ_upper_bytes=raw+(raw+511)//512+65536,
                recipe_JSON_planning_bytes=65536+256*t,
                filesystem_files=2, per_file_counter_overhead_bytes=8192,
                logical_int64_cut_plan_upper_bytes=89*t*h+h,
                durable_group_bound=k,durable_episode_bound=g)


def encode_v1(plan, metadata):
    import numpy as np
    plan_arrays = dict(
        representatives_u32=plan.representatives.detach().cpu().numpy(),
        episode_groups_u16=plan.episode_groups.detach().cpu().numpy(),
        episode_of_hold_u32=np.asarray(plan.episode_of_hold,dtype=np.int64),
        group_offsets_u32=np.asarray(plan.group_offsets,dtype=np.int64))
    arrays={}
    for name,value in plan_arrays.items():
        dtype=np.uint16 if name.endswith("u16") else np.uint32
        require(value.size>0 and (value>=0).all()
                and int(value.max())<=np.iinfo(dtype).max(),"Lossless verified integer range")
        arrays[name]=np.array(value,dtype=dtype,copy=True,order="C")
    arrays["above_horizon_bits"]=np.packbits(
        plan.above_horizon.cpu().numpy(),bitorder="little")
    arrays["root_saturated_bits"]=np.packbits(
        plan.root_saturated.cpu().numpy(),bitorder="little")
    k=int(plan.representatives.numel());g=int(plan.episode_groups.shape[0])
    limits=compact_upper_v1(plan.native_nodes,plan.native_holds,groups=k,episodes=g)
    receipt=dict(schema="H049_compact_cut_recipe_v1",format=FORMAT,
        native_nodes=plan.native_nodes,native_holds=plan.native_holds,
        haplotypes=plan.haplotypes,groups=k,episodes=g,
        source_fingerprints=dict(plan.source_fingerprints),cut_metadata=metadata,
        arrays={name:dict(dtype=str(a.dtype),shape=list(a.shape),bytes=int(a.nbytes))
                for name,a in arrays.items()},bounds=limits,
        logical_model_dtypes_preserved=True,observations_only=True,targets_read=False)
    require(sum(a.nbytes for a in arrays.values())==limits["compact_raw_array_bytes"],
            "Exact compact array accounting")
    return arrays,receipt


def decode_v1(payload, receipt, bank):
    import numpy as np
    import torch
    from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as m
    require(receipt["schema"]=="H049_compact_cut_recipe_v1"
            and receipt["format"]==FORMAT and receipt["observations_only"] is True
            and receipt["targets_read"] is False,"Original observation-only compact recipe")
    n,t,h=receipt["native_nodes"],receipt["native_holds"],receipt["haplotypes"]
    k,g=receipt["groups"],receipt["episodes"]
    require(bank.parent.device.type=="cpu" and bank.parent.numel()==n
            and tuple(bank.sample_index.shape)==(t,h) and h==952,
            "Exact original CPU native support")
    require(receipt["source_fingerprints"]=={
        name:m._fingerprint(getattr(bank,name)) for name in m._BIND_FIELDS},
        "Prepared cut recipe belongs to these exact native observations")
    limits=compact_upper_v1(n,t,h,groups=k,episodes=g)
    require(len(payload)<=limits["compressed_NPZ_upper_bytes"],
            "Bound compressed observation-only recipe")
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        require(set(z.files)==set(MEMBERS),"Exact compact archive members")
        arrays={}
        for name in MEMBERS:
            a=np.array(z[name],copy=True,order="C");d=receipt["arrays"][name]
            require(str(a.dtype)==d["dtype"] and list(a.shape)==d["shape"]
                    and int(a.nbytes)==d["bytes"],"Authenticated compact member dimensions")
            arrays[name]=a
    reps=arrays["representatives_u32"].astype(np.int64)
    ep=arrays["episode_groups_u16"].astype(np.int64)
    hold=arrays["episode_of_hold_u32"].astype(np.int64)
    offsets=arrays["group_offsets_u32"].astype(np.int64)
    require(reps.shape==(k,) and ep.shape==(g,h) and hold.shape==(t,)
            and offsets.shape==(t+1,) and offsets[0]==0 and offsets[-1]==k
            and (reps<n).all() and (ep<h).all() and (hold<g).all()
            and (np.diff(offsets)>0).all(),"Compact recipe geometry")
    require(hold[0]==0 and hold[-1]==g-1
            and np.isin(np.diff(hold),(0,1)).all(),"Ordered distinct-cut episodes")
    ep_counts=np.zeros((g,h),dtype=np.float32)
    for e,row in enumerate(ep):
        count=np.bincount(row,minlength=int(row.max())+1)
        require((count>0).all(),"Canonical nonempty local groups")
        ep_counts[e,:len(count)]=count
    leaf_groups=np.empty((t,h),dtype=np.int64);counts=[]
    for at,e in enumerate(hold):
        kk=int(ep[e].max())+1
        require(offsets[at+1]-offsets[at]==kk,"Exact native-hold group offsets")
        leaf_groups[at]=ep[e]+offsets[at]
        counts.extend(ep_counts[e,:kk].tolist())
    forward=m._direction_layouts(list(ep),range(g),h)
    reverse=m._direction_layouts(list(ep),range(g-1,-1,-1),h)
    above=np.unpackbits(arrays["above_horizon_bits"],bitorder="little")[:h].astype(bool)
    roots=np.unpackbits(arrays["root_saturated_bits"],bitorder="little")[:k].astype(bool)
    require(above.shape==(h,) and roots.shape==(k,),"Exact packed masks")
    def tensor(x,dtype=None):
        return torch.as_tensor(np.asarray(x),dtype=dtype).contiguous()
    plan=m.CutPlanV1(tensor(reps,torch.long),tensor(leaf_groups,torch.long),
        tensor(counts,torch.float32),tensor(ep,torch.long),tensor(ep_counts),
        *(tensor(x) for x in forward),*(tensor(x) for x in reverse),
        tensor(above,torch.bool),tensor(roots,torch.bool),tuple(map(int,offsets)),
        tuple(map(int,hold)),t,h,n,tuple(receipt["source_fingerprints"].items()),
        m._bank_stamps(bank),())
    plan=replace(plan,tensor_stamps=m._stamps(plan.tensors()))
    plan.validate(bank)
    require(plan.tensor_bytes()==receipt["cut_metadata"]["cut_plan_tensor_bytes"],
            "Lossless reconstructed logical PyG cut table bytes")
    return plan
